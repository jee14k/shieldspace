import { Injectable, UnauthorizedException } from '@nestjs/common';
// import { UsersService } from '../users/user.service';
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcrypt';
@Injectable()
export class AuthService {
  constructor(
    // private usersService: UsersService,
    private jwtService: JwtService,
  ) {}

  async signIn(email: string, pass: string): Promise<{ access_token: string }> {
    // const user = await this.usersService.findOne({ email });
    // if (!user || !(await bcrypt.compare(pass, user.password))) {
    //   throw new UnauthorizedException();
    // }
    if (email != process.env.EMAIL || pass != process.env.PASSWORD) {
      throw new UnauthorizedException();
    }
    const payload = { email: email, password: pass };
    return {
      access_token: await this.jwtService.signAsync(payload, {
        secret: process.env.JWT_SECRET,
      }),
    };
  }
}
